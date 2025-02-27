docker-up:
		$(DOCKER) compose -p $(PROJECT) --env-file ./config/.env.docker -f ./config/compose.yaml up -d

docker-down:
		$(DOCKER) compose -p $(PROJECT) -f ./config/compose.yaml down

docker-build:
		$(DOCKER) build -t $(USER)/$(PROJECT):$(VERSION) .

docker-run:
		$(DOCKER) container run --name $(PROJECT) -it  $(USER)/$(PROJECT):$(VERSION) /bin/bash

docker-gh-test:
		$(DOCKER) container run --name $(PROJECT) -p 3000:3000 --rm $(USER)/$(PROJECT):$(VERSION)

docker-gh-pull:
		$(DOCKER) pull ghcr.io/mohsenhariri/template-python:main 

docker-gh-run:
		$(DOCKER) run -p 3000:3000 -it --rm ghcr.io/mohsenhariri/template-python:main 