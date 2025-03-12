python src/main.py --kernel=1 --strideX=1 --strideY=1 > logs/1_1_1.log 2>&1

python src/main.py --kernel=11 --strideX=3 --strideY=3 > logs/11_3_3.log 2>&1

python src/main.py --kernel=25 --strideX=3 --strideY=3 > logs/25_3_3.log 2>&1
python src/main.py --kernel=49 --strideX=3 --strideY=3 > logs/49_3_3.log 2>&1

python src/main.py --kernel=25 --strideX=5 --strideY=5 > logs/25_5_5.log 2>&1
python src/main.py --kernel=49 --strideX=5 --strideY=5 > logs/49_5_5.log 2>&1

python src/main.py --kernel=25 --strideX=7 --strideY=7 > logs/25_7_7.log 2>&1
python src/main.py --kernel=49 --strideX=7 --strideY=7 > logs/49_7_7.log 2>&1 
