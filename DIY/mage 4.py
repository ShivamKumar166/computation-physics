import os
import time
import math
import matplotlib.pyplot as plt
from PIL import Image

# importing SVD helpers from my main module
from MAGE3 import svd_batch_fast, compress_to_jpeg, compute_metrics



#  Convert an image into R/G/B matrices

def image_to_rgb_matrices(path):
    """
    Load an image and split it into separate Red, Green, Blue matrices.
    Keeping channels separate makes it easier to run SVD on each one.

    Note: I’m returning w, h too because later I kept needing them.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    px = img.load()

    # Row-major: i loops over height, j loops over width
    R = [[px[j, i][0] for j in range(w)] for i in range(h)]
    G = [[px[j, i][1] for j in range(w)] for i in range(h)]
    B = [[px[j, i][2] for j in range(w)] for i in range(h)]
    return R, G, B, w, h




#  Singular values + energy plots

def compute_singular_values_for_channels(image_path, max_k=150):
    """
    Compute the top-k singular values for each RGB channel.
    I only really need the singular values, so I ignore U and Vᵀ.
    """
    R, G, B, w, h = image_to_rgb_matrices(image_path)

    # Each returns (U, S, Vt)
    # S is all we need here
    U_r, S_r, Vt_r = svd_batch_fast(R, max_k)
    U_g, S_g, Vt_g = svd_batch_fast(G, max_k)
    U_b, S_b, Vt_b = svd_batch_fast(B, max_k)

    return {"Red": S_r, "Green": S_g, "Blue": S_b}


def plot_singular_values(svals_dict, max_k=150):
    """
Log-scale plot showing how fast the singular values drop.
    If they decay quickly, image compresses well.
    """
    plt.figure(figsize=(8,5))
    for name, S in svals_dict.items():
        Splot = S[:max_k]
        plt.semilogy(range(1, len(Splot)+1), Splot, label=name)

    plt.xlabel("k")
    plt.ylabel("Singular value (log scale)")
    plt.title("Singular value decay (per RGB channel)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def energy_fraction(S):
    """
Cumulative fraction of total energy captured by first k singular values.
    Useful for visualizing how much "information" each rank contributes.


    """
    S2 = [s*s for s in S]
    total = sum(S2) or 1.0

    cum = []
    running = 0.0
    for v in S2:
        running += v
        cum.append(running / total)
    return cum


def plot_energy_capture(svals_dict, max_k=150):
    """
      Plot cumulative energy curves for each RGB channel.
    These curves help identify a reasonable cutoff for k.
    """
    plt.figure(figsize=(8,5))
    for name, S in svals_dict.items():
        ef = energy_fraction(S)
        # Only plot up to max_k for readability
        plt.plot(range(1, min(max_k, len(ef)) + 1), ef[:max_k], label=name)

    plt.xlabel("k")
    plt.ylabel("Energy captured")
    plt.title("Energy capture vs k")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()




#  Evaluation across different k values and jpeg_q values

def evaluate_k_values(image_path, k_values, jpeg_q=40, num_iter=2, remove_files=True):
    """
Try different SVD ranks (k) but keep JPEG quality fixed.
    The goal is to see how k affects PSNR, MSE, CR, runtime, etc.

    remove_files=True cleans up temporary JPEGs, so the folder doesn't fill up.
    """
    results = []  # each item: (k, mse, psnr, CR, runtime, comp_bytes)
    orig_bytes = os.path.getsize(image_path)

    for k in k_values:
        t0 = time.time()

        # Compress using the hybrid SVD+JPEG pipeline
        jpeg_path, runtime_comp = compress_to_jpeg(image_path, k=k,
                                                   num_iter=num_iter,
                                                   jpeg_q=jpeg_q)
        runtime_total = time.time() - t0

        orig_size, comp_size, CR, mse, psnr = compute_metrics(image_path, jpeg_path)

        results.append((k, mse, psnr, CR, runtime_total, comp_size))

        # Delete temp file unless debugging
        if remove_files:
            try:
                os.remove(jpeg_path)
            except:
                pass

    return results


def evaluate_jpeg_q_values(image_path, k_values, jpeg_q_values, num_iter=2, remove_files=True):
    """
For each SVD rank k, test multiple JPEG compression qualities.
    Returns:
        { k : [ (jpeg_q, mse, psnr, CR, runtime, comp_bytes), ... ] }
    """
    all_results = {}

    for k in k_values:
        k_results = []

        for q in jpeg_q_values:
            t0 = time.time()

            jpeg_path, runtime_comp = compress_to_jpeg(image_path, k=k,
                                                       num_iter=num_iter,
                                                       jpeg_q=q)
            runtime_total = time.time() - t0

            orig_size, comp_size, CR, mse, psnr = compute_metrics(image_path, jpeg_path)

            k_results.append((q, mse, psnr, CR, runtime_total, comp_size))

            if remove_files:
                try:
                    os.remove(jpeg_path)
                except:
                    pass

        all_results[k] = k_results

    return all_results




#  Plotting utilities

def plot_k_metrics(results):
    """
    Plot how MSE, PSNR, and CR change as k increases.
This gives a clear idea of diminishing returns.
    """
    ks = [r[0] for r in results]
    MSEs = [r[1] for r in results]
    PSNRs = [r[2] for r in results]
    CRs = [r[3] for r in results]

    # Bit cramped but manageable
    plt.figure(figsize=(15,4))

    plt.subplot(1,3,1)
    plt.plot(ks, MSEs, marker='o')
    plt.title("k vs MSE")
    plt.xlabel("k")
    plt.grid(True)

    plt.subplot(1,3,2)
    plt.plot(ks, PSNRs, marker='o')
    plt.title("k vs PSNR")
    plt.xlabel("k")
    plt.grid(True)

    plt.subplot(1,3,3)
    plt.plot(ks, CRs, marker='o')
    plt.title("k vs Compression Ratio")
    plt.xlabel("k")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_jpegq_curves(all_results, metric="psnr"):
    """
 Plot metric (PSNR, MSE, CR) against JPEG quality (1–95)
    for each SVD rank k.
    Useful to see which quality parameter gives acceptable results.
    """
    plt.figure(figsize=(8,5))

    for k, kr in all_results.items():
        qs = [x[0] for x in kr]

        if metric == "psnr":
            ys = [x[2] for x in kr]
            ylabel = "PSNR (dB)"
            title = "PSNR vs JPEG Quality"
        elif metric == "cr":
            ys = [x[3] for x in kr]
            ylabel = "Compression Ratio"
            title = "Compression Ratio vs JPEG Quality"
        elif metric == "mse":
            ys = [x[1] for x in kr]
            ylabel = "MSE"
            title = "MSE vs JPEG Quality"
        else:
            raise ValueError("Invalid metric name")

        plt.plot(qs, ys, marker='o', label=f"k={k}")

    plt.xlabel("JPEG Quality")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rate_distortion_from_results(results):
    """
 Plot PSNR vs Compression Ratio.
Basically a rate–distortion curve.
    """
    CRs = [r[3] for r in results]
    PSNRs = [r[2] for r in results]

    plt.figure(figsize=(6,4))
    plt.plot(CRs, PSNRs, marker='o')
    plt.xlabel("Compression Ratio")
    plt.ylabel("PSNR (dB)")
    plt.title("Rate–Distortion Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def find_optimal_k(results, psnr_gain_threshold=0.25):
    """
 Choose the smallest k after which PSNR gain becomes negligible.

    results is assumed sorted by increasing k.
    """
    for i in range(1, len(results)):
        gain = results[i][2] - results[i-1][2]
        if gain < psnr_gain_threshold:
            return results[i-1][0]

    return results[-1][0]


def recommend_jpeg_q_for_k(k_results, psnr_tol=0.5):
    """ Pick the lowest JPEG quality whose PSNR is within psnr_tol dB
    of the maximum achievable PSNR for this k.

    This avoids unnecessarily high q values.
    """
    best_psnr = max(x[2] for x in k_results)
    target = best_psnr - psnr_tol

    # Sort by q (ascending)
    k_results_sorted = sorted(k_results, key=lambda x: x[0])

    for q, mse, psnr, CR, rt, cb in k_results_sorted:
        if psnr >= target:
            return q, mse, psnr, CR, rt, cb

    # fallback if something unexpected happens
    return k_results_sorted[-1]




#  Full analysis workflow

def full_analysis(image_path,
                  k_values=[10,20,30,40,60,80,100,140],
                  jpeg_q_values=[10,20,30,40,50,60,70,80],
                  num_iter=2,
                  psnr_gain_threshold=0.25,
                  jpeg_psnr_tol=0.5):
    """
     Analysis that:
      1. Computes singular values and energy
      2. Tests different k values (rank)
      3. Chooses an optimal k
      4. Tests multiple JPEG qualities
      5. Recommends JPEG quality

    Basically, this ties all the smaller functions together.
    """
    print("==== SVD + JPEG Pipeline Analysis ====")
    print("Using image:", image_path)

    # --- Step 1: singular values ---
    print("\n1) Computing singular values (up to 150)...")
    svals = compute_singular_values_for_channels(image_path, max_k=150)
    plot_singular_values(svals, max_k=80)
    plot_energy_capture(svals, max_k=80)

    # --- Step 2: evaluate k values ---
    print("\n2) Evaluating k values (JPEG quality fixed at 40)...")
    res_k = evaluate_k_values(image_path, k_values, jpeg_q=40, num_iter=num_iter)
    plot_k_metrics(res_k)
    plot_rate_distortion_from_results(res_k)

    best_k = find_optimal_k(res_k, psnr_gain_threshold)
    print(f"\nRecommended k (PSNR gain < {psnr_gain_threshold} dB):", best_k)

    # --- Step 3: sweep JPEG quality ---
    print("\n3) Sweeping JPEG quality...")
    # Use best_k if available; otherwise use the first few k's
    chosen_k_values = [best_k] if best_k in k_values else k_values[:3]

    all_q_results = evaluate_jpeg_q_values(image_path,
                                           k_values=chosen_k_values,
                                           jpeg_q_values=jpeg_q_values,
                                           num_iter=num_iter)

    plot_jpegq_curves(all_q_results, metric="psnr")
    plot_jpegq_curves(all_q_results, metric="cr")

    # --- Step 4: recommend JPEG quality ---
    print("\n4) Recommending JPEG quality settings...")
    recs = {}

    for k, kr in all_q_results.items():
        qrec = recommend_jpeg_q_for_k(kr, psnr_tol=jpeg_psnr_tol)
        recs[k] = qrec

        q, mse, psnr, CR, rt, cb = qrec
        print(f"For k={k}: recommended jpeg_q={q}, PSNR={psnr:.2f} dB, CR={CR:.2f}, size={cb/1024:.2f} KB")

    # Optional final RD curve for one k
    some_k = next(iter(all_q_results))
    rd = [(x[3], x[2]) for x in all_q_results[some_k]]
    CRs, PSNRs = zip(*rd)

    plt.figure(figsize=(6,4))
    plt.plot(CRs, PSNRs, marker='o')
    plt.xlabel("Compression Ratio")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Rate–Distortion Curve for k={some_k}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nAnalysis complete.")

    return {
        "best_k": best_k,
        "jpeg_recommendations": recs,
        "k_results": res_k,
        "q_results": all_q_results
    }



if __name__ == "__main__":
    IMAGE = "sample-photo-2_imresizer.jpg"  # change this if needed
    out = full_analysis(IMAGE)
    print("\nFINAL RECOMMENDATIONS:", out["best_k"], out["jpeg_recommendations"])
