ABALONE = "abalone"
LOBSTER = "lobster"
FINFISH = "finfish"
SCALLOP = "scallop"
EPINEPHELUS_POLYPHEKADION = "epinephelus_polyphekadion_finfish"
LETHRINUS_OLIVACEUS = "lethrinus_olivaceus_finfish"

QUARTER = "quarter"
SQUARE = "square"
INCHES = "inches"
MM = "mm"
CM = "cm"

MIN_SIZE = 1.0
MAX_SIZE = 15.0


DEF_SQUARE_SIZE_IN = 2.0
DEF_SQUARE_SIZE_MM = 50.8

WIDTH_MEASUREMENT = "width";
LENGTH_MEASUREMENT = "length";

QUARTER_SIZE_MM = 24.26
QUARTER_SIZE_CM = 2.426

INCHES_TO_MM = 25.4
INCHES_TO_CM = 2.54


REF_TYPES = [ABALONE, QUARTER, SQUARE]
REF_UNITS = [INCHES, MM, CM]

def isFinfish(fishery_type):
    isFinfish = FINFISH in fishery_type
    return isFinfish

def isLobster(fishery_type):
    isLob = LOBSTER in fishery_type
    return isLob

def isScallop(fishery_type):
    isScall = SCALLOP in fishery_type
    return isScall

def isAbalone(fishery_type):
    isAb = ABALONE in fishery_type
    return isAb