import hashlib

from everyvoice.utils import slugify

BASENAME_MAX_LENGTH = 20


def truncate_basename(basename: str) -> str:
    """
    Shortens basename to BASENAME_MAX_LENGTH and uses the rest of basename to generate a sha1.
    This is done to make sure the file name stays short but that two utterances
    starting with the same prefix doesn't get ovverridden.
    """
    basename_cleaned = slugify(basename)
    if len(basename_cleaned) <= BASENAME_MAX_LENGTH:
        return basename_cleaned

    m = hashlib.sha1()
    m.update(bytes(basename, encoding="UTF-8"))
    return basename_cleaned[:BASENAME_MAX_LENGTH] + "-" + m.hexdigest()[:8]
