from typing import Annotated

from pydantic import Field, NonNegativeFloat, PositiveInt

from censor_engine.structs.censors import Censor

Percentage = float
BoundPercentage = Annotated[Percentage, Field(ge=0, le=1)]
MarginPercentage = Annotated[Percentage, Field(gt=-1)]

Seconds = NonNegativeFloat
Frames = PositiveInt

ListOfCensors = list[Censor]
Groups = list[list[str]]
