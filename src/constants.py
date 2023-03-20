AMINO_ACID_INDICES = {
    "-": 0,
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
}

AMINO_ACIDS = list(AMINO_ACID_INDICES.keys())
IMMUNOGENICITY_SCORES = [0, 1, 2]
START_TOKEN = ">"
IMMUNOGENICITY_ONE_HOT = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}

IMMUNOGENICITY_Q1 = 50.667
IMMUNOGENICITY_Q3 = 51.167
