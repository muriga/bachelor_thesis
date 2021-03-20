class Classifier:

    def train(self, path_owners: str, path_managers: str):
        """Prepare model if it have one."""
        pass

    def is_owner(self, path_to_pdf: str) -> bool:
        """Method tries to extract information from pdf, if it describes KUV who is owner."""
        pass

    def get_stop_words(self):
        with open(self.path_to_dataset + "stop-words.txt", encoding="utf-8") as f:
            lines = f.readlines()
        stop_words = {lines[i].replace('\n', ''): i for i in range(len(lines))}
        return stop_words
