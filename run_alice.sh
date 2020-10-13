echo "Starting alice.."

python training_alice.py > alice.log 2>&1 &
tail -f alice.log
