class Pack(dict):
    """
    Pack
    """
    def __getattr__(self, name):
        return self.get(name)

    def add(self, **kwargs):
        """
        Add items to `Pack` object
        """
        for k, v in kwargs.items():
            self[k] = v

    def flatten(self):
        """
        flatten
        """
        pack_list = []
        for vs in zip(*self.values()):
            pack = Pack(zip(self.keys(), vs))
            pack_list.append(pack)
        return pack_list

    def cuda(self, device=None):
        """
        cuda
        """
        pack = Pack()
        for k, v in self.items():
            if isinstance(v, tuple):
                pack[k] = tuple(x.cuda(device) for x in v)
            else:
                pack[k] = v.cuda(device)
        return pack
