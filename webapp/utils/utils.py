def split_array_equally(arr, number_of_element):
    return [arr[i:i+number_of_element] for i in range(len(arr))[::number_of_element]]
