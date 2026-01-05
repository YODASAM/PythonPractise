# Prime Number Generator (RNS-digit incremental filter, no division/mod)
# Improved version of Eric Olsen's approach:
#   - vectorized digit updates (no Python loops in hot path)
#   - defer "digit activation" until p^2 (so we only track primes <= sqrt(n))
#   - skip even candidates
# ---------------------------------------------------------------------------
# RNS-digit Prime Generator (no division/mod)
#
# Command-line examples:
#   # Generate the first 1000 primes (stats only)
#   python prime_generator_ver2.py 1000 --stats
#
#   # Generate the first 1000 primes (print primes to stdout)
#   python prime_generator_ver2.py 1000 --print
#
#   # Generate the first 1000 primes (print + stats + save to CSV)
#   python prime_generator_ver2.py 1000 --print --stats --file primes_1000.csv
# ---------------------------------------------------------------------------
import numpy as np
from numpy import asarray, savetxt
import argparse
import sys


def generate_primes_rns(count: int, collect_stats: bool = False):
    if count < 1:
        return [], {}

    # Always include 2
    primes = [2]
    if count == 1:
        return primes, {"square_multiplies": 0, "digit_increments": 0, "digit_wraps": 0, "zero_checks": 0}

    # Active moduli and their running residues for the CURRENT candidate n (odd n only).
    active_moduli = np.empty(0, dtype=np.int64)
    digits = np.empty(0, dtype=np.int64)

    # Primes are discovered in increasing order, so p^2 is also increasing.
    # We keep "pending" primes and a pointer to the next square where we activate a digit.
    pending_primes = []    # primes we have discovered but haven't activated yet
    pending_squares = []   # their squares
    next_square_idx = 0

    # Stats
    square_multiplies = 0
    digit_increments = 0
    digit_wraps = 0
    zero_checks = 0

    # We will iterate only odd candidates: 3, 5, 7, 9, ...
    n = 1
    step = 2

    while len(primes) < count:
        n += step  # next odd candidate

        # 1) Advance residues for currently active moduli by +2 (since we skip evens).
        if digits.size:
            digits += step
            digit_increments += digits.size

            # Wrap once is enough because step=2, so digits can exceed modulus by at most 2.
            wrap_mask = digits >= active_moduli
            if np.any(wrap_mask):
                digits[wrap_mask] -= active_moduli[wrap_mask]
                digit_wraps += int(np.count_nonzero(wrap_mask))

        # 2) Activate any prime whose square equals this candidate (n == p^2).
        #    When we activate at p^2, the residue is exactly 0, so we can append digit=0
        #    WITHOUT doing any division/mod.
        while next_square_idx < len(pending_squares) and pending_squares[next_square_idx] == n:
            p = pending_primes[next_square_idx]
            next_square_idx += 1

            active_moduli = np.append(active_moduli, p)
            digits = np.append(digits, 0)  # residue at n=p^2

        # 3) Prime test: if none of the active residues are 0, it's prime.
        #    (If there are no active moduli yet, everything looks prime until 9, which is fine:
        #     9 gets caught by activation of 3 at 3^2.)
        zero_checks += 1
        if digits.size and np.any(digits == 0):
            continue

        # If we got here, n is prime
        primes.append(n)

        # Schedule activation at n^2 (it will become active only when we reach that square).
        pending_primes.append(n)
        pending_squares.append(n * n)
        square_multiplies += 1

    stats = {}
    if collect_stats:
        stats = {
            "total_primes_found": len(primes),
            "square_multiplies": square_multiplies,
            "digit_increments": digit_increments,
            "digit_wraps": digit_wraps,
            "zero_checks": zero_checks,
            "active_moduli_final": int(active_moduli.size),
            "max_prime": primes[-1],
        }

    return primes, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cnt", type=int, help="the number of primes to generate starting with 2")
    parser.add_argument("-p", "--print", action="store_true", help="print primes to stdout")
    parser.add_argument("-s", "--stats", action="store_true", help="print statistics to stdout")
    parser.add_argument("-f", "--file", nargs=1, help="CSV formatted filename to save primes to")
    args = parser.parse_args()

    number = args.cnt
    if (number < 1) or (number > 1_000_000_000):
        sys.exit("Error: cnt argument should be between 1 and 1,000,000")

    print(f"Generating the first {number} primes (RNS-digit method, no division/mod):")
    print("Please wait !")

    primes, stats = generate_primes_rns(number, collect_stats=args.stats)

    if args.print:
        print("\n")
        print(np.array(primes, dtype=np.int64))

    if args.stats:
        print("\n")
        for k, v in stats.items():
            print(f"{k}: {v}")

    if args.file:
        filename = args.file[0]
        savetxt(filename, asarray(primes), delimiter=",", fmt="%d")
        print("\n")
        print(f"primes saved to {filename}")

    print("Primes Generation completed!")


if __name__ == "__main__":
    main()
