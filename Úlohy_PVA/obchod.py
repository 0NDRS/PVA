import sys

shelves = {} 
shopping_lists = []
current_shopping_list = []

adding = True
in_shopping_list_mode = False

print("Start adding items to shelves. To switch to shopping list mode, type 'list'. To end, type 'end'.")

while adding:
    if not in_shopping_list_mode:
        new_item = input("Add number of shelf or add item: ")
    
        if new_item == "end":
            adding = False
        elif new_item == "list":
            if current_shopping_list:
                shopping_lists.append(current_shopping_list)
                current_shopping_list = []
            in_shopping_list_mode = True
            print("Start adding items to shopping list. To start another shopping list, type 'list'. To end, type 'end'.")
        elif new_item.isdigit():
            current_shelf_id = int(new_item)
            shelves[current_shelf_id] = []
        elif new_item.strip() == "":
            print("Please enter a valid item to add to the list.")
        else:
            if 'current_shelf_id' in locals():
                shelves[current_shelf_id].append(new_item)
            else:
                print("First add the number of the shelf.")
    else:
        new_item = input("Add item to shopping list: ")
        if new_item == "end":
            if current_shopping_list:
                shopping_lists.append(current_shopping_list)
            adding = False
        elif new_item == "list":
            if current_shopping_list:
                shopping_lists.append(current_shopping_list)
                current_shopping_list = []
            print("Start adding items to new shopping list. To start another shopping list, type 'list'. To end, type 'end'.")
        elif new_item.strip() == "":
            print("Please enter a valid item for the shopping list.")
        else:
            current_shopping_list.append(new_item)

def print_shelves():
    for shelf_id, shelf in shelves.items():
        print(f"Shelf {shelf_id}: {shelf}")

def find_shelf_for_item(item):
    item_lower = item.lower()
    for shelf_id, shelf in shelves.items():
        for shelf_item in shelf:
            if item_lower in shelf_item.lower():
                return shelf_id, shelf_item
    return None, None

print("Content of shelves:")
print_shelves()

for list_index, shopping_list in enumerate(shopping_lists, start=1):
    results = []
    for item in shopping_list:
        shelf_id, shelf_item = find_shelf_for_item(item)
        if shelf_id is not None:
            results.append((shelf_id, item, shelf_item))
        else:
            results.append((None, item, None))

    sorted_results = sorted(results, key=lambda x: (x[0] is None, x[0]))

    print(f"\nShopping list {list_index} with shelf IDs:")
    for shelf_id, item, shelf_item in sorted_results:
        if shelf_id is not None:
            print(f"{item} is in shelf {shelf_id} ({shelf_item})")
        else:
            print(f"{item} was not found in any shelf.")
