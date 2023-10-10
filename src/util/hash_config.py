import hashlib


def hash_config(config, length: int = 20):
    """
    Hashes a config dictionary to a 20 character string.
    :param config: the config file to hash
    :return: hash of the config file
    """
    # Create a hashlib object (you can choose different hash algorithms like sha256, sha512, etc.)
    hasher = hashlib.sha256()

    # String to hash
    string_to_hash = str(config)

    # Update the hash object with the bytes of the string
    hasher.update(string_to_hash.encode('utf-8'))

    # Get the hexadecimal representation of the hash
    hash_result = hasher.hexdigest()[:length]
    return hash_result
