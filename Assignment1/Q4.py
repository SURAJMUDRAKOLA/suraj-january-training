# Q4.py
# Correctly removes even numbers from a list

def remove_even(nums):
    result = []
    for num in nums:
        if num % 2 != 0:
            result.append(num)
    return result


# 🔽 FUNCTION CALL (THIS WAS MISSING)
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(remove_even(nums))
