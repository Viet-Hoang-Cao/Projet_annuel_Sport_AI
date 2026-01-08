# Guide de Déploiement - SportAI

Ce guide explique comment déployer l'application SportAI avec Docker, Terraform et CI/CD.

## 🚀 Déploiement Local avec Docker

### Prérequis
- Docker et Docker Compose installés

### Étapes

1. **Construire et lancer l'application**:
   ```bash
   docker-compose up --build
   ```

2. **Accéder à l'application**:
   - Interface web: http://localhost:8000
   - API docs: http://localhost:8000/docs

3. **Arrêter l'application**:
   ```bash
   docker-compose down
   ```

## ☁️ Déploiement sur AWS avec Terraform

### Prérequis
- AWS CLI configuré avec les credentials appropriés
- Terraform >= 1.0 installé
- Permissions AWS pour créer VPC, ECS, ECR, etc.

### Étapes

1. **Configurer les variables Terraform**:
   ```bash
   cd terraform
   cp terraform.tfvars.example terraform.tfvars
   # Éditer terraform.tfvars avec vos valeurs
   ```

2. **Initialiser Terraform**:
   ```bash
   terraform init
   ```

3. **Vérifier le plan**:
   ```bash
   terraform plan
   ```

4. **Appliquer la configuration**:
   ```bash
   terraform apply
   ```

5. **Récupérer l'URL ECR**:
   ```bash
   terraform output ecr_repository_url
   ```


```
Projet_annuel_Sport_AI/
├── backend/              # Application FastAPI
│   ├── app/
│   │   ├── main.py       # Application principale
│   │   ├── training_plans.py  # Générateur de plans
│   │   └── detectors/    # Détecteurs d'exercices
│   └── static/           # Frontend HTML/CSS/JS
├── terraform/            # Infrastructure as Code
│   ├── main.tf          # Configuration principale
│   ├── variables.tf     # Variables
│   └── outputs.tf       # Sorties
├── .github/workflows/    # CI/CD pipelines
├── Dockerfile           # Image Docker
└── docker-compose.yml   # Compose pour développement
```



### Docker
```bash
# Build
docker build -t sportai .

# Run
docker run -p 8000:8000 sportai

# Logs
docker-compose logs -f
```

### Terraform
```bash
# Format
terraform fmt

# Validate
terraform validate

# Destroy
terraform destroy
```

### Git
```bash
# Créer et pusher la branche hachem
git checkout -b hachem
git add .
git commit -m "feat: ..."
git push -u origin hachem
```
