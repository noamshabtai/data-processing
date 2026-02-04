import argparse


def main():
    parser = argparse.ArgumentParser(description="Stock Analyzer CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--symbol", required=True, help="Stock symbol")
    train_parser.add_argument("--period", default="1y", help="Training period")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--output", required=True, help="Output model path")

    predict_parser = subparsers.add_parser("predict", help="Run predictions")
    predict_parser.add_argument("--symbol", required=True, help="Stock symbol")
    predict_parser.add_argument("--model", required=True, help="Model path")
    predict_parser.add_argument("--simulate", action="store_true", help="Simulate mode")

    args = parser.parse_args()

    if args.command == "train":
        print(f"Training model for {args.symbol} with period {args.period}")
    elif args.command == "predict":
        print(f"Running predictions for {args.symbol}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
