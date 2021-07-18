server.run:
	python main.py

playground:
	python simple_request.py

curl:
	curl -X POST -F image=@data/lynx.jpg 'http://localhost:5000/api/predict'

curl.italian:
	curl -X POST -F image=@data/53_1xg.jpg 'http://localhost:5000/api/italian-docs/predict'