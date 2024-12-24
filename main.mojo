fn main():
    var person: Person = Person("Ryan")
    person.printDatShit()

struct Person:
    var name: String

    fn __init__(out self, name: String, completion: () -> Void ):
        self.name = name

    fn printDatShit(self):
        print(self.name)