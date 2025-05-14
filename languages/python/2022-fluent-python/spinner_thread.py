import itertools
import time
from threading import Thread, Event


def spin(msg, done: Event) -> None:
    for char in itertools.cycle(r"\|/"):
        status = f"\r{char} {msg}"
        print(status, end="", flush=True)
        if done.wait(0.01):
            break
    blanks = " " * len(status)
    print(f"\r{blanks}\r", end="")


def slow() -> int:
    time.sleep(10)
    return 42


def supervisor() -> int:
    done = Event()
    spinner = Thread(target=spin, args=("Thinking!", done))
    print(f"spinner object: {spinner}\n")
    spinner.start()
    # print("Spinner is started!")
    result = slow()
    done.set()
    spinner.join()
    return result


def main() -> None:
    result = supervisor()
    print(f"Answer: {result}")


if __name__ == "__main__":
    main()
