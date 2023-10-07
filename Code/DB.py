import os

class DB:
    def __init__(self, db_name):
        self.db_name = db_name
        if not os.path.exists('mockDB'):
            os.makedirs('mockDB')
        self.db = open('mockDB\\' + db_name + '.txt', 'a+')
    
    def add(self, key, data):
        if self.find(key) == None:
            self.db.write(key + ':' + data + '\n')

    def find(self, key):
        self.db.seek(0)
        for line in self.db.readlines():
            if line.split(':')[0] == key:
                return line.split(':')[1]
        return None

    def read(self):
        self.db.seek(0)
        return self.db.read()
    
    def clear(self):
        self.db.seek(0)
        self.db.truncate(0)
    
    def close(self):
        self.db.close()
