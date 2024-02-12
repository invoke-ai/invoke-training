import uvicorn

from invoke_training.ui.app import build_app


def main():
    app = build_app()
    uvicorn.run(app)


if __name__ == "__main__":
    main()
