try:
    import skopt
    print("Successfully imported skopt")
except ImportError as e:
    print(f"Error importing skopt: {e}")

try:
    from skopt import something
    print("Successfully imported from skopt")
except ImportError as e:
    print(f"Error importing from skopt: {e}")

try:
    import scikit_opt
    print("Successfully imported scikit_opt")
except ImportError as e:
    print(f"Error importing scikit_opt: {e}")

try:
    from scikit_opt import something
    print("Successfully imported from scikit_opt")
except ImportError as e:
    print(f"Error importing from scikit_opt: {e}")

try:
    import skpot
    print("Successfully imported skpot")
except ImportError as e:
    print(f"Error importing skpot: {e}") 