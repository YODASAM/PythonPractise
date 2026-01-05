# 05_bank_account_with_5_accounts_test.py

class InsufficientFundsError(Exception):
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Cannot withdraw ₹{amount}. Only ₹{balance} available.")

class BankAccount:
    def __init__(self, owner, initial_balance=0):
        self.owner = owner
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative")
        self.balance = initial_balance

    def deposit(self, amount):
        if not isinstance(amount, (int, float)):
            raise TypeError("Deposit amount must be a number")
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount
        print(f"  ✓ Deposited ₹{amount}. New balance: ₹{self.balance}")

    def withdraw(self, amount):
        if not isinstance(amount, (int, float)):
            raise TypeError("Withdrawal amount must be a number")
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise InsufficientFundsError(self.balance, amount)
        self.balance -= amount
        print(f"  ✓ Withdrew ₹{amount}. New balance: ₹{self.balance}")

    def show_balance(self):
        print(f"  Account: {self.owner} | Balance: ₹{self.balance}")

# List of 5 test accounts with different starting balances
accounts_data = [
    ("Alice", 1000),
    ("Bob", 500),
    ("Charlie", 0),
    ("Diana", 2000),
    ("Eve", 150)
]

print("Bank Account System - Testing 5 Accounts\n" + "="*60)

for owner, initial_balance in accounts_data:
    print(f"\nCreating account for {owner} with initial balance ₹{initial_balance}")
    try:
        account = BankAccount(owner, initial_balance)
        account.show_balance()

        # Mix of valid and invalid operations
        operations = [
            ("deposit", 300),           # Valid
            ("withdraw", 100),          # Valid (for most)
            ("deposit", -50),           # Invalid: negative deposit
            ("withdraw", 5000),         # Likely invalid: too large
            ("deposit", "abc"),         # Invalid: wrong type
            ("withdraw", 200),          # May succeed or fail depending on balance
        ]

        for action, amount in operations:
            try:
                if action == "deposit":
                    account.deposit(amount)
                elif action == "withdraw":
                    account.withdraw(amount)
            except InsufficientFundsError as e:
                print(f"  ✗ Transaction failed: {e}")
            except ValueError as e:
                print(f"  ✗ Invalid operation: {e}")
            except TypeError as e:
                print(f"  ✗ Type error: {e}")
            except Exception as e:
                print(f"  ✗ Unexpected error: {type(e).__name__}: {e}")

        account.show_balance()  # Final balance

    except ValueError as e:
        print(f"  ✗ Account creation failed: {e}")
    except Exception as e:
        print(f"  ✗ Unexpected setup error: {e}")

    print("-" * 60)
