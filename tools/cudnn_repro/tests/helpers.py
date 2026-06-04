def tensor_list(tensors):
    return [{"tid": int(tid), **entry} for tid, entry in sorted(tensors.items(), key=lambda item: int(item[0]))]
