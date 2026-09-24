from app.services.cv_service import CVService


def main():
    service = CVService()

    try:
        service.start()

        print("CV service started.")
        print("Press Ctrl+C to stop.")

        while service.running:
            pass

    except KeyboardInterrupt:
        print("\nStopping CV service...")

    finally:
        service.stop()


if __name__ == "__main__":
    main()