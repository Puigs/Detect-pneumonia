# Creation d'user sur le vps et configuration des comptes

### Configuration vps
<p>
Sur le vps : 54.38.185.154, ont été créé différent comptes pour l'équipe de IA de montpellier, composé de Brayan, Manon, Clément et Gab.
Ce document que vous vous appreter à lire permet justement de créé différents users sur un vps et parle de la configuration du vps.

Tout d'abord il faudra créér un groupe d'utilisateur, permettant de set aux différents membres du groupes des règles/accès.

Pour créer un groupe : 
```bash
groupadd <name>
```

Pour créer un user : 
```bash
sudo adduser <name>
```

Pour ajouter un user à un groupe:
```bash
sudo usermod -a -G <name_group> <name_user>
```

Le but de créer un groupe est de permette aux différents users de pouvoir travailler sur le même dossier.
Pour cela on va créer un dossier dans /home que l'on va appeler local ou share par exemple.
Puis nous allons modifier les droits du groupe afin que chaque utilisateur puisse accéder/modifier/executer les différents fichier:
```bash
sudo chgrp -R <name_group> <path-to-dir>
sudo chmod -R 777 <path-to-dir>
```

Puis nous allons utiliser la commande suivante afin que les nouveaux fichiers généré soit tout de même accesible et modifiable:
```bash
sudo find <path-to-dir> -type d -exec chmod 2775 {} \;    
```
</p>

### Configuration user

<p>
Pour chaque user, nous allons installer quelques outils comme zsh et conda.

Pour installer zsh il vous suffit d'effectuer la commande suivante : 
```bash
sudo apt-get install zsh
```

Pour installer oh-my-zsh il vous faut effectuer la commande suivante :
```bash
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

Pour installer conda il vous executer le script "Anaconda3-2021.11-Linux-x86_64.sh" en bash, puis executer la commande suivante pour que cela fonctionne en zsh : 
```bash
conda init zsh
```
</p>