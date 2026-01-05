import numpy as np
primes = np.array([2])         # start with the first prime
#print("starting primes array:", primes)
digits = np.array([0])         # start with first digit equal to zero


def contains_zero(array):
    for j in range(0, array.size):
        if array[j] == 0:
            return True
    return False
        
def wrap_digits(primes, digits):
    for i in range(primes.size):
        if(digits[i] == primes[i]):
            digits[i] = 0

num_of_multiplies = 0
num_of_increments = 0
            
num_primes = 1
last = 2
number = 1000;   # NUMBER OF PRIMES STARTING AT 2 THAT WILL BE FOUND BY THE RESIDUE NUMBER SIEVE
print("Generating the first", number, "primes without any division:")
print("Please wait !")
while(num_primes < number):
    index = 0
    max_prime = primes[primes.size-1] ** 2   # MAX TRIAL DIGIT WILL BE LAST PRIME FOUND SQUARED
    num_of_multiplies += 1
    #for trial in range(last+1, max_prime):
    for trial in range(last + 1 + (last % 2), max_prime, 2):
#        print("trial:", trial)
        digits += 1     # increment the digits array
        num_of_increments += digits.size
        wrap_digits(primes, digits)    # wrap the digits array against each digit's moduli (the primes array)
#        print("digits:", digits)      # THIS IS THE NATURAL RESIDUE NUMBER
        if not contains_zero(digits):
            primes = np.insert(primes, primes.size, trial)
            digits = np.insert(digits, digits.size, 0)
            print("%d " % trial, end='')              # this prints out each prime as its found, comment out if not needed
#            print("new primes array:", primes)       # THE MODULI OF THE RESIDUE NUMBER IS EXTENDED BY THE NEW PRIME!
#            print("new digits array:", digits)       # THE NEW DIGIT FOR THE NEW MODULUS INSERTED IS ALWAYS ZERO!
            num_primes += 1
            if(num_primes > number):
                break
    last = max_prime-1
    
print("\n")
print(primes)
print("\n")
print("total number of multiplications:", num_of_multiplies)
print("total number of increments (and compares:)", num_of_increments)

