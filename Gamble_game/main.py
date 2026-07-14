import random
import time


START_BALANCE = 1000
MIN_BET = 10


def pause(seconds=0.7):
    time.sleep(seconds)


def line():
    print("-" * 48)


def ask_int(prompt, minimum=None, maximum=None):
    while True:
        value = input(prompt).strip()
        if not value.isdigit():
            print("Введите целое число.")
            continue

        number = int(value)
        if minimum is not None and number < minimum:
            print(f"Число должно быть не меньше {minimum}.")
            continue
        if maximum is not None and number > maximum:
            print(f"Число должно быть не больше {maximum}.")
            continue
        return number


def ask_bet(balance):
    max_bet = balance
    return ask_int(f"Ставка ({MIN_BET}-{max_bet}): ", MIN_BET, max_bet)


def card_value(card):
    if card in ["J", "Q", "K"]:
        return 10
    if card == "A":
        return 11
    return int(card)


def hand_score(hand):
    score = sum(card_value(card) for card in hand)
    aces = hand.count("A")

    while score > 21 and aces:
        score -= 10
        aces -= 1

    return score


def draw_card():
    return random.choice(["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"])


def show_hand(owner, hand, hide_first=False):
    if hide_first:
        cards = ["?"] + hand[1:]
        print(f"{owner}: {' '.join(cards)}")
    else:
        print(f"{owner}: {' '.join(hand)} | очки: {hand_score(hand)}")


def slots(balance):
    line()
    print("Слоты: соберите 3 одинаковых символа.")
    bet = ask_bet(balance)
    symbols = ["7", "BAR", "CHERRY", "LEMON", "BELL"]

    print("Крутим барабаны...")
    pause()
    result = [random.choice(symbols) for _ in range(3)]
    print(" | ".join(result))

    if result[0] == result[1] == result[2]:
        multiplier = 8 if result[0] == "7" else 5
        win = bet * multiplier
        print(f"Джекпот! Вы выиграли {win}.")
        return balance + win

    if len(set(result)) == 2:
        win = bet * 2
        print(f"Две одинаковые картинки. Вы выиграли {win}.")
        return balance + win

    print(f"Не повезло. Вы потеряли {bet}.")
    return balance - bet


def roulette(balance):
    line()
    print("Рулетка: число 0-36, цвет или четность.")
    print("1. Угадать число: выплата x35")
    print("2. Красное/черное: выплата x2")
    print("3. Четное/нечетное: выплата x2")
    choice = ask_int("Ваш выбор: ", 1, 3)
    bet = ask_bet(balance)

    number = random.randint(0, 36)
    red_numbers = {
        1, 3, 5, 7, 9, 12, 14, 16, 18,
        19, 21, 23, 25, 27, 30, 32, 34, 36,
    }
    color = "красное" if number in red_numbers else "черное"
    if number == 0:
        color = "зеленое"

    if choice == 1:
        guess = ask_int("Ваше число (0-36): ", 0, 36)
        print("Шарик летит...")
        pause()
        print(f"Выпало: {number} ({color})")
        if guess == number:
            win = bet * 35
            print(f"Точное попадание! Вы выиграли {win}.")
            return balance + win
    elif choice == 2:
        print("1. Красное")
        print("2. Черное")
        guess = ask_int("Ваш цвет: ", 1, 2)
        wanted = "красное" if guess == 1 else "черное"
        print("Шарик летит...")
        pause()
        print(f"Выпало: {number} ({color})")
        if wanted == color:
            win = bet * 2
            print(f"Цвет угадан! Вы выиграли {win}.")
            return balance + win
    else:
        print("1. Четное")
        print("2. Нечетное")
        guess = ask_int("Ваш выбор: ", 1, 2)
        wanted_even = guess == 1
        print("Шарик летит...")
        pause()
        print(f"Выпало: {number} ({color})")
        if number != 0 and (number % 2 == 0) == wanted_even:
            win = bet * 2
            print(f"Угадали! Вы выиграли {win}.")
            return balance + win

    print(f"Ставка проиграла. Вы потеряли {bet}.")
    return balance - bet


def blackjack(balance):
    line()
    print("Блэкджек: набери ближе к 21, чем дилер.")
    bet = ask_bet(balance)

    player = [draw_card(), draw_card()]
    dealer = [draw_card(), draw_card()]

    while True:
        line()
        show_hand("Дилер", dealer, hide_first=True)
        show_hand("Вы", player)

        if hand_score(player) == 21:
            print("У вас 21!")
            break
        if hand_score(player) > 21:
            print(f"Перебор. Вы потеряли {bet}.")
            return balance - bet

        print("1. Взять карту")
        print("2. Остановиться")
        choice = ask_int("Ваш ход: ", 1, 2)
        if choice == 1:
            player.append(draw_card())
        else:
            break

    print("Ход дилера...")
    pause()
    while hand_score(dealer) < 17:
        dealer.append(draw_card())
        pause(0.4)

    line()
    show_hand("Дилер", dealer)
    show_hand("Вы", player)

    player_score = hand_score(player)
    dealer_score = hand_score(dealer)

    if dealer_score > 21 or player_score > dealer_score:
        win = bet * 2
        print(f"Победа! Вы выиграли {win}.")
        return balance + win
    if player_score == dealer_score:
        print("Ничья. Ставка возвращается.")
        return balance

    print(f"Дилер победил. Вы потеряли {bet}.")
    return balance - bet


def print_menu(balance):
    line()
    print("CASINO NIGHT")
    print(f"Баланс: {balance}")
    print("1. Слоты")
    print("2. Рулетка")
    print("3. Блэкджек")
    print("4. Правила")
    print("5. Выйти")


def rules():
    line()
    print("Правила:")
    print("- Минимальная ставка: 10.")
    print("- Если баланс закончился, игра завершится.")
    print("- Слоты: 3 одинаковых символа дают x5, три семерки дают x8.")
    print("- Рулетка: число платит x35, цвет и четность платят x2.")
    print("- Блэкджек: победа платит x2, ничья возвращает ставку.")


def main():
    balance = START_BALANCE
    print("Добро пожаловать в CASINO NIGHT!")

    while balance >= MIN_BET:
        print_menu(balance)
        choice = ask_int("Выберите игру: ", 1, 5)

        if choice == 1:
            balance = slots(balance)
        elif choice == 2:
            balance = roulette(balance)
        elif choice == 3:
            balance = blackjack(balance)
        elif choice == 4:
            rules()
        else:
            break

    line()
    if balance < MIN_BET:
        print("Баланс меньше минимальной ставки. Игра окончена.")
    print(f"Итоговый баланс: {balance}")
    print("Спасибо за игру!")


if __name__ == "__main__":
    main()
