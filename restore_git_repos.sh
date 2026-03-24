
# Repository: papers/rejected/2023aamas_bluesky/supplementary
echo -e "${BLUE}Restoring: papers/rejected/2023aamas_bluesky/supplementary${NC}"
if [ -d "papers/rejected/2023aamas_bluesky/supplementary/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "papers/rejected/2023aamas_bluesky"
    
    # Clone the repository
    if git clone "git@github.com:yasserfarouk/gbmaaai2023.git" "papers/rejected/2023aamas_bluesky/supplementary"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"
        
        # Checkout the original branch if not already on it
        cd "papers/rejected/2023aamas_bluesky/supplementary"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "main" ]; then
            if git checkout "main" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: main"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: main"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi


# Repository: papers/rejected/2023aamas_bluesky/supplementary
echo -e "${BLUE}Restoring: papers/rejected/2023aamas_bluesky/supplementary${NC}"
if [ -d "papers/rejected/2023aamas_bluesky/supplementary/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "papers/rejected/2023aamas_bluesky"
    
    # Clone the repository
    if git clone "git@github.com:yasserfarouk/gbmaaai2023.git" "papers/rejected/2023aamas_bluesky/supplementary"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"
        
        # Checkout the original branch if not already on it
        cd "papers/rejected/2023aamas_bluesky/supplementary"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "main" ]; then
            if git checkout "main" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: main"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: main"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi


# Repository: papers/rejected/2023aamas_bluesky/supplementary
echo -e "${BLUE}Restoring: papers/rejected/2023aamas_bluesky/supplementary${NC}"
if [ -d "papers/rejected/2023aamas_bluesky/supplementary/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "papers/rejected/2023aamas_bluesky"
    
    # Clone the repository
    if git clone "git@github.com:yasserfarouk/gbmaaai2023.git" "papers/rejected/2023aamas_bluesky/supplementary"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"
        
        # Checkout the original branch if not already on it
        cd "papers/rejected/2023aamas_bluesky/supplementary"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "main" ]; then
            if git checkout "main" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: main"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: main"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi


# Repository: papers/rejected/2023aamas_bluesky/supplementary
echo -e "${BLUE}Restoring: papers/rejected/2023aamas_bluesky/supplementary${NC}"
if [ -d "papers/rejected/2023aamas_bluesky/supplementary/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "papers/rejected/2023aamas_bluesky"
    
    # Clone the repository
    if git clone "git@github.com:yasserfarouk/gbmaaai2023.git" "papers/rejected/2023aamas_bluesky/supplementary"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"
        
        # Checkout the original branch if not already on it
        cd "papers/rejected/2023aamas_bluesky/supplementary"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "main" ]; then
            if git checkout "main" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: main"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: main"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi

