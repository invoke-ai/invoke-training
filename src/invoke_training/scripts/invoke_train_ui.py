import argparse

import uvicorn

from invoke_training.ui.app import build_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="The server host. Set `--host 0.0.0.0` to make the app available on your network.",
    )
    parser.add_argument("--port", default=8000, type=int, help="The server port.")
    args = parser.parse_args()

    app = build_app()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
