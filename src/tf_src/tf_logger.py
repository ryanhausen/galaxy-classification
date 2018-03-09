import tensorflow as tf

from colorama import init, Fore
init(autoreset=True)

red = lambda s: Fore.RED + str(s) + Fore.RESET
blue = lambda s: Fore.BLUE + str(s) + Fore.RESET
yellow = lambda s: Fore.YELLOW + str(s) + Fore.RESET
green = lambda s: Fore.GREEN + str(s) + Fore.RESET

def info(msg):
    tf.logging.info(green(msg))

def debug(msg):
    tf.logging.debug(yellow(msg))

def error(msg):
    tf.logging.error(red(msg))