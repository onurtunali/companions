import itertools
import time
from multiprocessing import Process, Event
from multiprocessing import synchronize


def spin(mesg: str, done: synchronize.Event):
    for char in itertools.cycle(r"\|/"):
        status = f"\r{char} {mesg}"
        print(status, end="", flush=True)
        if done.wait(0.01):
            break
    blanks = " " * len(status)
    print(f"\r{blanks}\r", end="")


def slow():
    time.sleep(4)
    return 42


def supervisor():
    done = Event()
    spinner = Process(target=spin, args=("Thinking!", done))
    print(f"spinner object: {spinner}")
    spinner.start()
    result = slow()
    done.set()
    spinner.join()
    return result


def main():
    result = supervisor()
    print(f"Answer: {result}")


if __name__ == "__main__":
    main()
