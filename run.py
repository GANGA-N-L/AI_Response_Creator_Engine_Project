import asyncio
import sys

# Must run before uvicorn creates its event loop - psycopg3's async driver
# refuses to run under Windows' default ProactorEventLoop, and by the time
# `uvicorn app.main:app` imports the app module, that loop already exists.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", reload=True)
