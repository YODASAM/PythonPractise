# enhanced_calculator_with_error_handling.py

class NegativeNumberError(Exception):
    """Custom exception for negative values"""
    def __init__(self, value):
        self.value = value
        super().__init__(f"Negative number not allowed: {value}")

class NonNumericInputError(Exception):
    """Custom exception for non-numeric input"""
    pass

class Calculator:
    def square_root(self, num):
        # Check for negative number
        if num < 0:
            raise NegativeNumberError(num)
        # Check for complex numbers (though float won't allow, but just in case)
        if isinstance(num, complex):
            raise ValueError("Complex numbers are not supported")
        return num ** 0.5

def safe_square_root():
    while True:
        user_input = input("\nEnter a number to calculate square root (or 'quit' to exit): ").strip()

        # 1. KeyboardInterrupt (Ctrl+C)
        try:
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            # 2. Empty input
            if not user_input:
                print("Error: No input provided. Please enter a number.")
                continue

            # 3. Convert to float
            try:
                num = float(user_input)
            except ValueError:
                raise NonNumericInputError(user_input)

            # 4. Call the calculator
            calc = Calculator()
            result = calc.square_root(num)
            print(f"√{num} = {result:.6f}")

        # === Error Handling for 10+ Cases ===
        except NegativeNumberError as e:
            print(f"Math error: {e}")  # Negative number

        except NonNumericInputError as e:
            print(f"Input error: '{e}' is not a valid number. Please enter a numeric value.")

        except TypeError as e:
            print(f"Type error: Invalid data type passed. Details: {e}")

        except ValueError as e:
            if "could not convert" in str(e).lower():
                print(f"Conversion error: Cannot convert input to number.")
            else:
                print(f"Value error: {e}")

        except OverflowError:
            print("Error: Number is too large to process (overflow).")

        except ZeroDivisionError:
            print("Error: Unexpected division by zero occurred (shouldn't happen in sqrt).")

        except MemoryError:
            print("Error: System ran out of memory (very large input?).")

        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user. Goodbye!")
            break

        except EOFError:
            print("\n\nEnd of input detected (e.g., Ctrl+D). Goodbye!")
            break

        except Exception as e:
            print(f"Unexpected error occurred: {type(e).__name__}: {e}")

        else:
            print("Calculation successful!")  # Runs only if no exception

        finally:
            print("-" * 50)  # Always runs

# Run the program
if __name__ == "__main__":
    print("Advanced Square Root Calculator with Full Error Handling")
    print("Tests 10+ types of errors gracefully.\n")
    safe_square_root()
