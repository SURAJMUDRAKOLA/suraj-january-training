# Q2.py
# This function safely tracks order history across calls

def add_order(order_id, orders=None):
    """
    Accepts an order ID and stores it in a list.
    Uses None as default to avoid mutable default argument issues.
    """
    if orders is None:
        orders = []

    orders.append(order_id)
    return orders


# Sample calls
print(add_order(101))
print(add_order(102))
print(add_order(103))
