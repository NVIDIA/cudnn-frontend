def tensor_list(tensors):
    return [entry for _, entry in sorted(tensors.items(), key=lambda item: int(item[0]))]
