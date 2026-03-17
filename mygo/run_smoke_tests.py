import os
import unittest


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(repo_root, "tests")
    suite = unittest.defaultTestLoader.discover(start_dir=tests_dir, pattern="test*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()

