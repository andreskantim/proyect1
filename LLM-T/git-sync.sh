#!/bin/bash
# Script para sincronizar cambios con GitHub de forma automática
# Uso: ./git-sync.sh "mensaje del commit"

set -e  # Salir si hay algún error

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Git Sync - Sincronización Automática ===${NC}\n"

# Verificar que estamos en el directorio correcto
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: No estás en un repositorio git${NC}"
    exit 1
fi

# Obtener el mensaje del commit
if [ -z "$1" ]; then
    # Si no se proporciona mensaje, usar uno genérico con timestamp
    COMMIT_MSG="Auto-update: $(date '+%Y-%m-%d %H:%M:%S')"
else
    COMMIT_MSG="$1"
fi

echo -e "${BLUE}1. Verificando cambios...${NC}"
git status --short

echo -e "\n${BLUE}2. Añadiendo archivos al staging area...${NC}"
git add .

# Verificar si hay cambios para commitear
if git diff --staged --quiet; then
    echo -e "${GREEN}✓ No hay cambios para commitear${NC}"
    exit 0
fi

echo -e "\n${BLUE}3. Creando commit...${NC}"
git commit -m "$COMMIT_MSG

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"

echo -e "\n${BLUE}4. Sincronizando con GitHub...${NC}"
BRANCH=$(git branch --show-current)

# Intentar hacer push
if git push origin "$BRANCH" 2>/dev/null; then
    echo -e "\n${GREEN}✓ Cambios sincronizados exitosamente con GitHub${NC}"
else
    echo -e "\n${RED}Error al hacer push. Puede que necesites:${NC}"
    echo -e "  1. Crear el repositorio en GitHub primero"
    echo -e "  2. Configurar tus credenciales: git config credential.helper store"
    echo -e "  3. Hacer el primer push manualmente: git push -u origin $BRANCH"
    exit 1
fi

echo -e "\n${GREEN}✓ Sincronización completada${NC}"
