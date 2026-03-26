"""Allow running as: python -m harness.runner"""
from harness.runner import main
import asyncio

asyncio.run(main())
