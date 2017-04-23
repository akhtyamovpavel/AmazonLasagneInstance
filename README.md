# AmazonLasagneInstance
Мануал по поднятию инстанса на Amazon со свежей версией Lasagne на g2.2xlarge

## Вместо предисловия
Обычно бывает сложно поднять инстанс на Амазоне, особенно с поддержкой запуска на видеокарте. Есть специальные образы, но за них необходимо платить. Поэтому ниже предлагается мануал, каким образом можно установить Jupyter Notebook с фреймворком Lasagne.

## Создание инстанса
Будем создавать инстанс на сайте aws.amazon.com (если нет студенческой лицензии, то ее можно запросить доступ через AWS Educate, дают обычно около 100$).
<b>Предупреждение</b> Час работы на инстансе в регионе US составляет порядка 0.65$, поэтому используйте ресурсы с умом.
### Шаги по созданию инстанса:
1. После авторизации на сайте, вы попадаете на главный экран борды, выбрав пункт EC2. Далее необходимо нажать на кнопку "Launch Instance".
2. На шаге выбора инстанса выбираем слева пункт "Community AMI", вводим в поиск "nvidia cuda", выбираем "amzn-ami-2016.03.e-amazon-ecs-optimized-nvidia-cuda", на этом инстансе гарантирован корректный запуск.
3. Выбираем тип инстанса - g2.2xlarge, нажимаем "review and launch" (можно покопаться с доп. настройками, но выбираем не спотовый инстанс)
4. Нажимаем на кнопку "Launch", Amazon предложит создать ssh-ключ или выбрать из предложенных. При выборе нового ключа необходимо ввести ключ и сохранить приватный ключ.
5. Инстанс начал подниматься.

## Получение доступа к инстансу и установка зависимостей
1. Переходим в меню выбора инстансов (View instances)
2. В таблице появится инстанс типа g2.2xlarge, выбираем его. Снизу появится меню с описанием доступов к инстансу. Нажимаем на клавишу Connect. Копируем самую нижнюю команду и подсоединяемся по ssh (необходимо только root заменить на ec2-user). Если вы - пользовать Windows, то для соединения необходимо скачать программу Putty, ссылка для скачивания: https://the.earth.li/~sgtatham/putty/latest/w64/putty.exe
3. После коннекта на сервер, ставим anaconda (Python 3 или Python2 - версия не важна). Ссылки для скачивания:
```
wget https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh # Python 3
wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh # Python 2
```
4. Установка Anaconda:
```
bash Anaconda3-4.3.1-Linux-x86_64.sh # Для Python 3, для Python 2 аналогично
```
Желательно не прописывать в переменную PATH Anaconda, чтобы проблем с совместимостью библиотек не было.

5. Устанавливаем Vim:
```
sudo yum install vim
```
6. Прописываем следующие строки в `.bashrc` (выполняем команду `vim .bashrc`):
```
export PATH="/usr/local/cuda-7.5/bin:$PATH"
export LIBRARY_PATH="/usr/local/cuda-7.5/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH"
```
Это подключит библиотеки CUDA. Для того, чтобы добавить строки, нажмите на `i`, после введения строк нажмите на `Esc`, а затем `:wq` и `Enter`, чтобы сохранить изменения.

7. Обновляем изменения, прописываем команду `source .bashrc`.

8. Выкачиваем библиотеку cudnn, которая требует lasagne, для этого идем на сайт NVIDIA (https://developer.nvidia.com/cudnn), нажимаем на кнопку Download (если предлагает зарегистрироваться, то жмем на кнопку "Join Now" и регистрируемся, заполняем опросы). Заходим на сайт, качаем "cuDNN v5.1 for CUDA 7.5, Linux"

9. Открываем терминал, выполняем аналогичную команду для заливки файлов на сервер (команду лучше сделать из той папки, где лежит ваш скачанный ключ):
```
scp -i "key.pem" cudnn-7.5-linux-x64-v5.1.tgz ec2-user@ec2-xx-xxx-xx-xx.us-west-2.compute.amazonaws.com:~/
```
, где key.pem - ключ соединения к серверу.

10. Распаковываем на сервере данные и раскладываем по папкам:
```
tar -zxvf cudnn-7.5-linux-x64-v5.1.tgz
cd cuda
sudo cp include/cudnn.h /usr/local/cuda-7.5/include/
sudo cp lib64/libcudnn.so /usr/local/cuda-7.5/lib64/
sudo cp lib64/libcudnn.so.5 /usr/local/cuda-7.5/lib64/
sudo cp lib64/libcudnn.so.5.1.10 /usr/local/cuda-7.5/lib64/
sudo cp lib64/libcudnn_static.a /usr/local/cuda-7.5/lib64/
cd
```

11. Устанавливаем Lasagne:
```
~/anaconda3/bin/pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
~/anaconda3/bin/pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

12. Прописываем файл `.theanorc` (содержание можно найти в `.theanorc.example` в репозитории)
