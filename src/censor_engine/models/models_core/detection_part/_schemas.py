from enum import StrEnum


class PartNameType(StrEnum):
    NAME = "name"
    ID_AND_NAME = "id_name"
    ID_AND_NAME_AND_MERGED = "id_name_merged"
    NAME_AND_MERGED = "name_merged"
