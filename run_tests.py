"""
Script to run all tests with different configurations
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print results"""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70)
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    """Run test suite"""
    print("="*70)
    print("WEATHER ML PROJECT - TEST SUITE")
    print("="*70)
    
    all_passed = True
    
    # 1. Run all tests
    if not run_command("pytest tests/ -v", "All Tests"):
        all_passed = False
    
    # 2. Run only API tests
    if not run_command("pytest tests/test_api.py -v", "API Tests"):
        all_passed = False
    
    # 3. Run only model tests
    if not run_command("pytest tests/test_models.py -v", "Model Tests"):
        all_passed = False
    
    # 4. Run with coverage (optional)
    print("\n" + "="*70)
    print("Running: Tests with Coverage Report (optional)")
    print("="*70)
    subprocess.run("pytest tests/ --cov=app --cov=ml --cov-report=term --cov-report=html", shell=True)
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())