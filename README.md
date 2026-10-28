# Практическая работа по основам нейронных сетей

NumPy MLP для бинарной классификации, CLI подготовки изображений и параллельная аугментация. Все команды выполняются из корня проекта. Python 3.10 или новее.

## Установка на macOS

Откройте Терминал и перейдите в распакованную папку: введите `cd `, перетащите папку в окно Терминала и нажмите Enter.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_data.py
```

На Windows: `py -m venv .venv`, затем в PowerShell `.venv\Scripts\Activate.ps1`.

## Часть 1

```bash
python -m part1_mlp.app.main
```

Вводите ответы по порядку, каждый с новой строки:

```text
1
2 8 1
sigmoid
2
data/input/xor.csv
0.2
3
0.3
1000
16
4
m
0 0;0 1;1 0;1 1
5
data/output/model.npz
6
data/output/model.npz
4
f
data/input/predict.csv
7
0
```

Пункт 7 открывает график; закройте его окно для возврата в меню. Ожидаемые классы: 0, 1, 1, 0. Пункт 1 создаёт новую сеть; сменить активацию можно через него, выбрав `relu`. Последний слой всегда 1 (бинарная классификация), скрытых слоёв может быть несколько. CSV числовой, с заголовком; последний столбец — 0 или 1. CSV предсказаний содержит только признаки с заголовком. Сохранение включает веса, смещения, активацию, историю и параметры стандартизации.

Сделайте скриншоты: (1) создание сети и загрузка CSV; (2) ошибка обучения и test accuracy; (3) четыре предсказания и сохранение/загрузка. На Mac: Shift+Command+4. Не используйте скриншоты чужого запуска.

## Часть 2

```bash
python -m part2_prepare.app.cli prepare --path data/input/images --ext png --size 32 --output data/output/png32.npy
python -m part2_prepare.app.cli prepare --path data/input/images --ext png jpg --size 64 --output data/output/prepared.npy
python -m part2_prepare.app.cli doctor
```

Скриншоты: обе команды prepare со статистикой; doctor с версиями. Первый результат: (24, 32, 32, 3); второй: (48, 64, 64, 3). CUDA для проекта не требуется. Изображения синтетические, не внешняя обучающая база.

Настройки: аргумент CLI имеет приоритет над ENV, затем используются значения по умолчанию. `.env.example` — образец переменных; автоматического чтения .env нет. На Mac можно выполнить:

```bash
export DATA_PATH=data/input/images
export OUTPUT_DIR=data/output
export LOG_LEVEL=INFO
```

Расширения указываются через пробел: `--ext png jpg`, не через запятую. Часть 2 поддерживает изображения; CSV обрабатывается в части 1. Это допустимый выбор из условия «изображения или CSV».

## Часть 3

```bash
python -m part3_parallel.app.performance --path data/input/images --readers 4 --processes 2
cat data/output/performance_report.json
```

Сделайте скриншоты лога потоков и JSON. В Windows вместо `cat` можно использовать `Get-Content`. На 48 небольших изображениях процессы часто медленнее из-за старта и передачи массивов. Значение `outputs_equal` должно быть `true`; результаты имеют форму (144, 64, 64, 3). Время зависит от компьютера — обновите значения в отчёте своим результатом.

Для более тяжёлого эксперимента (потребует больше памяти):

```bash
python -m part3_parallel.app.performance --size 256 --readers 4 --processes 2
```

Честное сравнение: повторить каждую конфигурацию несколько раз; первый последовательный запуск может прогревать файловый кеш для параллельного. Текущий JSON содержит один эксперимент, не статистически устойчивый бенчмарк. Очереди и число ожидающих задач ограничены, но финальные массивы остаются в памяти. Это учебная реализация, не потоковый датасет неограниченного размера.

## Проверка

```bash
python verify.py
```

Проверяется решение XOR, равенство предсказаний после загрузки и численные градиенты весов и смещений для sigmoid/ReLU. В `logs/` приложен реальный вывод проверки из среды подготовки. Это текстовые протоколы, не скриншоты вашего компьютера.

Для дополнительной проверки стиля:

```bash
python -m pip install flake8
python -m flake8 part1_mlp part2_prepare part3_parallel generate_data.py verify.py --max-line-length=100
```

## Git и сдача

1. Установите Git, если команда `git --version` не работает. На Mac система предложит Command Line Tools.
2. На github.com создайте новый пустой репозиторий `neural-network-practice`, без автоматического README. Выберите видимость по требованиям преподавателя. Публикацию выполняете вы.
3. Задайте своё имя и email для коммитов (строки ниже — образцы, замените значения):

```bash
git init
git config user.name "Ваше имя"
git config user.email "Ваш email"
git add .gitignore .env.example requirements.txt README.md generate_data.py data/input part1_mlp verify.py
git commit -m "Implement MLP with CSV loading and model persistence"
git add part2_prepare
git commit -m "Add image preparation CLI and environment diagnostics"
git add part3_parallel
git commit -m "Add threaded reading and process-based augmentation"
git add report.docx
git commit -m "Add practice report"
git branch -M main
```

Затем выполните команды `git remote add origin ...` и `git push -u origin main`, которые GitHub покажет для вашего репозитория. Вставляйте реальный URL своего репозитория. Файл отчёта добавляйте после заполнения. Результаты в data/output и logs исключены из Git; при необходимости приложите их отдельно в LMS.

## Что осталось сделать лично

- В Word открыть `report.docx`: вписать ФИО, направление и реальную ссылку на репозиторий.
- Запустить команды и вставить собственные скриншоты в отмеченные места, удалить подсказки в квадратных скобках.
- В разделе замеров заменить время на собственные результаты. Пример из среды подготовки явно подписан.
- Сохранить отчёт, выполнить коммиты и отправить проект на GitHub.
- В LMS прикрепить отчёт, ссылку и при необходимости архив исходников.

## Частые ошибки

`No module named part1_mlp`: вы не в корне проекта, выполните `pwd` и `ls`.
`No module named numpy`: активируйте .venv и установите requirements.txt.
`File not found`: запустите generate_data.py и проверьте путь.
`CSV features do not match network`: для XOR первый слой равен 2.
`nvidia-smi unavailable`: нормальная ситуация на Mac, расчёты идут на CPU.
