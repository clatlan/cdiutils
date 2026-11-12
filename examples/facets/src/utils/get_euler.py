#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A standalone script to compute Bunge (ZXZ) Euler angles from a 3x3
rotation matrix stored in a text file.
"""

import numpy as np
import argparse
import sys

def rotation_matrix_to_zxz_euler(matrix):
    """
    Converts a 3x3 rotation matrix to Bunge (ZXZ) Euler angles.
    This function handles the gimbal lock singularity.

    Args:
        matrix (np.ndarray): The 3x3 rotation matrix.

    Returns:
        tuple: A tuple of (phi1, Phi, phi2) in radians.
    """
    R = np.array(matrix)

    # Check for gimbal lock singularity
    if not np.isclose(R[2, 2], 1.0) and not np.isclose(R[2, 2], -1.0):
        # General case (no gimbal lock)
        phi1 = np.arctan2(R[2, 0], -R[2, 1])
        Phi = np.arccos(R[2, 2])
        phi2 = np.arctan2(R[0, 2], R[1, 2])
    else:
        # Gimbal lock case
        if np.isclose(R[2, 2], 1.0):
            # Phi is 0, convention sets phi2 to 0
            phi1 = np.arctan2(R[0, 1], R[0, 0])
            Phi = 0.0
            phi2 = 0.0
        else: # R[2, 2] is -1.0
            # Phi is 180 degrees, convention sets phi2 to 0
            phi1 = np.arctan2(-R[0, 1], -R[0, 0])
            Phi = np.pi
            phi2 = 0.0

    return phi1, Phi, phi2

def main():
    """
    Main function to parse arguments and run the calculation.
    """
    parser = argparse.ArgumentParser(
        description="Compute Bunge (ZXZ) Euler angles from a rotation matrix file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'input_file',
        type=str,
        help="Path to the text file containing the 3x3 rotation matrix."
    )

    args = parser.parse_args()

    try:
        # Load the matrix from the text file
        orientation_matrix = np.loadtxt(args.input_file)

        # Validate the shape of the loaded matrix
        if orientation_matrix.shape != (3, 3):
            print(f"Error: The input file '{args.input_file}' does not contain a 3x3 matrix.", file=sys.stderr)
            sys.exit(1)

        # Calculate Euler angles
        phi1, Phi, phi2 = rotation_matrix_to_zxz_euler(orientation_matrix)

        # Print the results
        print("Bunge (ZXZ) Euler Angles:")
        print(f"  phi1: {phi1:.3f} rad")
        print(f"  Phi:  {Phi:.3f} rad")
        print(f"  phi2: {phi2:.3f} rad")

    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
