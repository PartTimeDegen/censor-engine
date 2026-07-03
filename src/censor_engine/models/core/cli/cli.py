from ._parser import build_parser
from ._processor import process_arguments
from .structs import ParsedArguments


def get_cli_inputs() -> ParsedArguments:
    """
    This function is used to get and process the arguments/cli for
    censor_engine.

    Returns:
        Parsed Arguments

    """
    parser = build_parser()
    args, _ = parser.parse_known_args()
    return process_arguments(args=args)
