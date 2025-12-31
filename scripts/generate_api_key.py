#!/usr/bin/env python3
"""
Generate cryptographically secure API keys for the LLM inference server.
"""

import secrets
import string


def generate_api_key(prefix: str = "sk", length: int = 32) -> str:
    """
    Generate a cryptographically secure API key.

    Args:
        prefix: Prefix for the API key (default: "sk")
        length: Length of the random portion (default: 32)

    Returns:
        Generated API key in format: {prefix}-{random_string}
    """
    # Use URL-safe characters for the random portion
    alphabet = string.ascii_letters + string.digits
    random_string = ''.join(secrets.choice(alphabet) for _ in range(length))

    return f"{prefix}-{random_string}"


def main():
    """Generate and display API keys."""
    print("=" * 60)
    print("LLM Inference Server - API Key Generator")
    print("=" * 60)
    print()

    # Generate regular API key
    api_key = generate_api_key(prefix="sk", length=32)
    print("Regular API Key (for user access):")
    print(f"  {api_key}")
    print()

    # Generate admin API key
    admin_key = generate_api_key(prefix="sk-admin", length=32)
    print("Admin API Key (for management endpoints):")
    print(f"  {admin_key}")
    print()

    print("=" * 60)
    print("Instructions:")
    print("=" * 60)
    print("1. Copy the keys above")
    print("2. Edit your .env file")
    print("3. Set API_KEYS with one or more regular keys (comma-separated)")
    print("4. Set ADMIN_API_KEY with the admin key")
    print()
    print("Example .env configuration:")
    print(f"  API_KEYS={api_key}")
    print(f"  ADMIN_API_KEY={admin_key}")
    print()
    print("For multiple user keys:")
    print(f"  API_KEYS={api_key},{generate_api_key()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
