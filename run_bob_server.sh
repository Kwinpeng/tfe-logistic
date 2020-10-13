echo "Starting bob.."
python training_bob.py > bob.log 2>&1 &
sleep 5
cat bob.log
echo "Starting server.."
python training_server.py > server.log 2>&1 &
sleep 5
cat server.log
