import time, math, os
from PIL import Image
import matplotlib.pyplot as plt


# ================================================================
#                      SVD HELPER FUNCTIONS
# ================================================================
# an approximate SVD based on power iterations.

def transpose(A):
    """
    Return the transpose of A (m×n -> n×m).
    Used when we need Aᵀ in the SVD steps.
    """
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def matmul_fast(A, B, block=32):
    """
    Blocked matrix multiply: C = A * B
    - A: m×n, B: n×p
    - Returns m×p matrix
    Block size helps performance on larger matrices (cache locality).
    """
    m, n, p = len(A), len(A[0]), len(B[0])
    C = [[0.0] * p for _ in range(m)]

    # weird-looking loops but they help speed on big data
    for ii in range(0, m, block):
        for kk in range(0, n, block):
            for jj in range(0, p, block):
                for i in range(ii, min(ii + block, m)):
                    for k in range(kk, min(kk + block, n)):
                        aik = A[i][k]  # small micro-optimization
                        for j in range(jj, min(jj + block, p)):
                            C[i][j] += aik * B[k][j]
    return C


def normalize_columns(A):
    """
Normalize columns of A to unit length.
 Important step to keep iterates stable during power iterations.
    """
    m, n = len(A), len(A[0])
    for j in range(n):
        norm = math.sqrt(sum(A[i][j] ** 2 for i in range(m))) + 1e-12
        for i in range(m):
            A[i][j] /= norm
    return A


def gram_schmidt(A):
    """
Classical Gram-Schmidt orthonormalization applied to columns.
     It is not the most stable variant, but paired with normalization
     it works fine in our approximate SVD loop.
    """
    m, n = len(A), len(A[0])
    Q = [[0.0] * n for _ in range(m)]

    for j in range(n):
        # copy column j
        for i in range(m):
            Q[i][j] = A[i][j]

        # subtract projection onto earlier basis vectors
        for k in range(j):
            dot = sum(Q[i][k] * Q[i][j] for i in range(m))
            for i in range(m):
                Q[i][j] -= dot * Q[i][k]

        # normalize the resulting vector
        norm = math.sqrt(sum(Q[i][j] ** 2 for i in range(m))) + 1e-12
        for i in range(m):
            Q[i][j] /= norm

    return Q


def svd_batch_fast(A, k=80, num_iter=10):
    """
Approximate truncated SVD using power iteration on AᵀA.
    A: input matrix (m×n)
     k: number of components to compute
     num_iter: refinement iterations

    Returns: (U, S, Vt) where U is m×k, S is list len k, Vt is k×n
    """
    m, n = len(A), len(A[0])

    # deterministic pseudo-random init (keeps runs reproducible)
    def lcgg(seed):
        a = 1103515245
        c = 12345
        m = 32768
        while True:
            seed = (a * seed + c) % m
            yield seed / m

    # generate n*k random numbers
    rng = lcgg(seed=123)  # any fixed seed for reproducibility
    rand = [next(rng) for _ in range(n * k)]

    # build initial V matrix
    V = [[rand[i * k + j] - 0.5 for j in range(k)] for i in range(n)]

    # normalize first
    V = normalize_columns(V)

    for it in range(num_iter):
        AV = matmul_fast(A, V)
        AtAV = matmul_fast(transpose(A), AV)

        # alternate between normalization and Gram-Schmidt
        if it % 2 == 1:
            V = gram_schmidt(AtAV)
        else:
            V = normalize_columns(AtAV)

    # get U (approx) and S
    AV = matmul_fast(A, V)
    S = []
    for j in range(k):
        col = [AV[i][j] for i in range(m)]
        s = math.sqrt(sum(x * x for x in col))
        S.append(s)

        # normalize columns to form U
        for i in range(m):
            AV[i][j] /= s + 1e-12

    return AV, S, transpose(V)


def reconstruct(U, S, Vt):
    """
     Reconstruct the rank-k approximation: U * diag(S) * Vt.
    Implemented manually for control and to avoid external deps.
    """
    m, n = len(U), len(Vt[0])
    k = len(S)

    out = [[0.0] * n for _ in range(m)]

    for r in range(k):
        for i in range(m):
            ui = U[i][r] * S[r]
            for j in range(n):
                out[i][j] += ui * Vt[r][j]

    return out



#                       SVD + JPEG WORKFLOW

# This section applies the SVD approximation to image channels and
# then encodes the reconstructed image using JPEG for additional
# compression. Keeps final saving as JPEG because pure SVD images
# are large and inconvenient.


def compress_to_jpeg(input_path, k=100, num_iter=8, jpeg_q=50):
    """
    Hybrid compress:
      compute SVD on Red channel
      use same basis for Green & Blue
      reconstruct all channels and save as JPEG
    """
    start_time = time.time()

    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    px = img.load()

    # split channels into 2D lists
    R = [[px[j, i][0] for j in range(w)] for i in range(h)]
    G = [[px[j, i][1] for j in range(w)] for i in range(h)]
    B = [[px[j, i][2] for j in range(w)] for i in range(h)]

    # SVD on red channel only (time-saving approximation)
    U_r, S_r, Vt = svd_batch_fast(R, k=k, num_iter=num_iter)
    V = transpose(Vt)

    # project green and blue onto same subspace
    U_g = matmul_fast(G, V)
    U_b = matmul_fast(B, V)

    # scale by inverse singular values to get coefficients
    inv_sig = [1.0 / (s + 1e-12) for s in S_r]
    for col in range(k):
        for i in range(h):
            U_g[i][col] *= inv_sig[col]
            U_b[i][col] *= inv_sig[col]

    # reconstruct approximations
    Rk = reconstruct(U_r, S_r, Vt)
    Gk = reconstruct(U_g, S_r, Vt)
    Bk = reconstruct(U_b, S_r, Vt)

    # pack into output image (clamp to valid 0..255)
    out = Image.new("RGB", (w, h))
    data = []
    for i in range(h):
        for j in range(w):
            data.append((
                int(max(0, min(255, Rk[i][j]))),
                int(max(0, min(255, Gk[i][j]))),
                int(max(0, min(255, Bk[i][j]))),
            ))
    out.putdata(data)

    # final JPEG write
    jpeg_path = f"svd_output_q{jpeg_q}.jpg"
    out.save(jpeg_path, format="JPEG", quality=jpeg_q, optimize=True, progressive=True)

    runtime = time.time() - start_time
    return jpeg_path, runtime



#                          METRICS
# ================================================================
# Compute standard compression quality metrics:
# Compression Ratio (CR)
#  Mean Squared Error (MSE)
#  Peak Signal-to-Noise Ratio (PSNR)


def compute_metrics(original, compressed):
    """
    Compare original and compressed images and return:
      orig_size, comp_size, CR, mse, psnr
    """
    orig_size = os.path.getsize(original)
    comp_size = os.path.getsize(compressed)

    CR = orig_size / comp_size if comp_size > 0 else float("inf")

    # ensure same size for comparison (use smaller one)
    img1 = Image.open(original).convert("RGB")
    img2 = Image.open(compressed).convert("RGB")
    w = min(img1.width, img2.width)
    h = min(img1.height, img2.height)
    img1 = img1.resize((w, h))
    img2 = img2.resize((w, h))
    p1 = list(img1.getdata())
    p2 = list(img2.getdata())

    # MSE across RGB channels
    mse = sum(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
        for a, b in zip(p1, p2)
    ) / (3 * w * h)

    psnr = float("inf") if mse == 0 else 10 * math.log10((255 ** 2) / mse)

    return orig_size, comp_size, CR, mse, psnr



#    VISUAL COMPARISON

def show_comparison(original, jpeg_path, metrics, runtime):
    """
    Display original and compressed images side-by-side and print metrics.
    Simple helper for quick visual checks.
    """
    orig_size, comp_size, CR, mse, psnr = metrics

    print("\n=========== METRICS ===========")
    print(f"Original Size       : {orig_size / 1024:.2f} KB")
    print(f"Compressed Size     : {comp_size / 1024:.2f} KB")
    print(f"Compression Ratio   : {CR:.3f}")
    print(f"MSE                 : {mse:.5f}")
    print(f"PSNR                : {psnr:.2f} dB")
    print(f"Runtime             : {runtime:.3f} s")
    print("================================\n")

    img0 = Image.open(original)
    img1 = Image.open(jpeg_path)

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # show original
    ax[0].imshow(img0)
    ax[0].set_title(f"Original\n{orig_size / 1024:.2f} KB", fontsize=14)
    ax[0].axis("off")

    # show compressed
    ax[1].imshow(img1)
    ax[1].set_title(f"Compressed\n{comp_size / 1024:.2f} KB", fontsize=14)
    ax[1].text(
        0.5, -0.05,
        f"CR={CR:.2f}\nMSE={mse:.6f}\nPSNR={psnr:.2f} dB\nRuntime={runtime:.3f}s",
        ha='center', va='top', fontsize=12, transform=ax[1].transAxes
    )
    ax[1].axis("off")

    plt.subplots_adjust(bottom=0.25)
    plt.show()



#                             MAIN LOOP

# Interactive

def main():
    while True:
        img = "sample-photo-2_imresizer.jpg"

        k = int(input("Enter k (SVD rank): ") or "80")
        num_iter = 2  # keep small for speed; increase for slightly better accuracy
        jpeg_q = int(input("Enter jpeg_quality (1–95): ") or "50")

        jpeg_path, runtime = compress_to_jpeg(img, k, num_iter, jpeg_q)
        metrics = compute_metrics(img, jpeg_path)
        show_comparison(img, jpeg_path, metrics, runtime)


if __name__ == "__main__":
    main()
