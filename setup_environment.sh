#!/bin/bash

# ============================================================================
# EPSS ML Pipeline Environment Setup Script
# ============================================================================
# This script sets up the complete environment for the EPSS ML pipeline:
# - Creates Python virtual environment
# - Installs Python dependencies
# - Sets up Java (OpenJDK 17) for PySpark
# - Optionally installs Hadoop
# - Configures environment variables
#
# Usage: ./setup_environment.sh [--with-hadoop]
# Make executable: chmod +x setup_environment.sh
# ============================================================================

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="epss-env"
JAVA_VERSION="17"
REQUIREMENTS_FILE="requirements.txt"

# Parse command line arguments
INSTALL_HADOOP=false
if [[ $# -gt 0 && "$1" == "--with-hadoop" ]]; then
    INSTALL_HADOOP=true
    echo -e "${CYAN}Hadoop installation requested${NC}"
fi

# Logging functions
log() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run a step with error handling
run_step() {
    local step_name="$1"
    local description="$2"
    shift 2
    
    log "Starting: ${BLUE}${step_name}${NC}"
    log "Description: ${description}"
    echo "----------------------------------------"
    
    if "$@"; then
        success "${step_name} completed successfully"
        echo ""
    else
        error "${step_name} failed"
        error "Setup stopped. Please check the error messages above."
        exit 1
    fi
}

# Step functions
setup_python_venv() {
    log "Setting up Python virtual environment: ${VENV_NAME}"
    
    # Check if Python3 is available
    if ! command_exists python3; then
        error "Python3 is not installed. Please install Python3 first."
        exit 1
    fi
    
    # Create virtual environment
    if [[ -d "$VENV_NAME" ]]; then
        warning "Virtual environment ${VENV_NAME} already exists. Removing it..."
        rm -rf "$VENV_NAME"
    fi
    
    python3 -m venv "$VENV_NAME"
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    success "Virtual environment created and activated"
}

install_python_dependencies() {
    log "Installing Python dependencies"
    
    # Ensure we're in the virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        error "Virtual environment not activated"
        exit 1
    fi
    
    # Check if requirements.txt exists
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        error "Requirements file not found: $REQUIREMENTS_FILE"
        exit 1
    fi
    
    # Install requirements
    pip install -r "$REQUIREMENTS_FILE"
    
    # Install PySpark explicitly (in case it's not in requirements.txt)
    log "Installing PySpark..."
    pip install pyspark
    
    success "Python dependencies installed"
}

setup_java() {
    log "Setting up Java (OpenJDK ${JAVA_VERSION})"
    
    # Update package list
    sudo apt-get update
    
    # Install OpenJDK 17
    sudo apt-get install -y "openjdk-${JAVA_VERSION}-jdk-headless"
    
    # Set up alternatives
    local java_path="/usr/lib/jvm/java-${JAVA_VERSION}-openjdk-amd64"
    
    sudo update-alternatives --install /usr/bin/java java "${java_path}/bin/java" 1
    sudo update-alternatives --install /usr/bin/javac javac "${java_path}/bin/javac" 1
    
    # Set JAVA_HOME
    export JAVA_HOME="$java_path"
    
    # Add to bashrc if not already present
    if ! grep -q "export JAVA_HOME=" ~/.bashrc; then
        echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc
        log "Added JAVA_HOME to ~/.bashrc"
    fi
    
    # Verify Java installation
    java -version
    
    success "Java setup completed"
}

setup_hadoop() {
    if [[ "$INSTALL_HADOOP" != "true" ]]; then
        info "Skipping Hadoop installation (not requested)"
        return
    fi
    
    log "Setting up Hadoop"
    
    # Install Hadoop
    sudo apt-get update
    sudo apt-get install -y hadoop-native
    
    # Set up Hadoop environment variables
    export HADOOP_HOME="/usr/lib/hadoop"
    export LD_LIBRARY_PATH="${HADOOP_HOME}/lib/native:${LD_LIBRARY_PATH:-}"
    
    # Add to bashrc if not already present
    if ! grep -q "export HADOOP_HOME=" ~/.bashrc; then
        echo "export HADOOP_HOME=$HADOOP_HOME" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$HADOOP_HOME/lib/native:\$LD_LIBRARY_PATH" >> ~/.bashrc
        log "Added Hadoop environment variables to ~/.bashrc"
    fi
    
    success "Hadoop setup completed"
}

verify_installation() {
    log "Verifying installation"
    
    # Check Java
    if ! command_exists java; then
        error "Java installation verification failed"
        exit 1
    fi
    
    # Check Python environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        error "Virtual environment not activated"
        exit 1
    fi
    
    # Check PySpark
    if ! python -c "import pyspark; print('PySpark version:', pyspark.__version__)" 2>/dev/null; then
        error "PySpark installation verification failed"
        exit 1
    fi
    
    # Check if we can create a Spark session
    if ! python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.appName('test').getOrCreate(); spark.stop()" 2>/dev/null; then
        error "Spark session creation failed"
        exit 1
    fi
    
    success "All components verified successfully"
}

create_activation_script() {
    log "Creating activation script"
    
    cat > activate_epss.sh << 'EOF'
#!/bin/bash
# EPSS ML Pipeline Environment Activation Script
# Usage: source activate_epss.sh

# Activate virtual environment
source epss-env/bin/activate

# Set Java environment (adjust path if needed)
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Set Hadoop environment (if installed)
if [ -d "/usr/lib/hadoop" ]; then
    export HADOOP_HOME=/usr/lib/hadoop
    export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native:$LD_LIBRARY_PATH
fi

echo "🚀 EPSS ML Pipeline environment activated!"
echo "Java version: $(java -version 2>&1 | head -n 1)"
echo "Python version: $(python --version)"
echo "PySpark version: $(python -c 'import pyspark; print(pyspark.__version__)' 2>/dev/null || echo 'Not available')"
echo ""
echo "Ready to run the pipeline with: ./ml_pipeline/run_pipeline.sh"
EOF

    chmod +x activate_epss.sh
    success "Activation script created: activate_epss.sh"
}

# Main setup function
main() {
    log "${CYAN}🚀 Starting EPSS ML Pipeline Environment Setup${NC}"
    log "This script will set up:"
    log "1. Python virtual environment (${VENV_NAME})"
    log "2. Python dependencies from ${REQUIREMENTS_FILE}"
    log "3. PySpark"
    log "4. Java (OpenJDK ${JAVA_VERSION})"
    if [[ "$INSTALL_HADOOP" == "true" ]]; then
        log "5. Hadoop (requested)"
    fi
    echo "========================================"
    echo ""
    
    # Check if we're in the right directory
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        error "Requirements file not found: $REQUIREMENTS_FILE"
        error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Run setup steps
    run_step "PYTHON-VENV" "Create and activate Python virtual environment" setup_python_venv
    run_step "PYTHON-DEPS" "Install Python dependencies and PySpark" install_python_dependencies
    run_step "JAVA-SETUP" "Install and configure Java (OpenJDK ${JAVA_VERSION})" setup_java
    run_step "HADOOP-SETUP" "Install and configure Hadoop (optional)" setup_hadoop
    run_step "VERIFICATION" "Verify all installations" verify_installation
    run_step "ACTIVATION-SCRIPT" "Create environment activation script" create_activation_script
    
    echo "========================================"
    success "${GREEN}🎉 Environment Setup Completed Successfully!${NC}"
    echo ""
    log "Next steps:"
    log "1. Activate the environment: source activate_epss.sh"
    log "2. Run the ML pipeline: ./ml_pipeline/run_pipeline.sh"
    echo ""
    warning "Note: You may need to restart your terminal or run 'source ~/.bashrc' for environment variables to take effect."
}

# Check if script is run with sudo for certain operations
if [[ $EUID -eq 0 ]]; then
    error "This script should not be run as root (sudo will be used when needed)"
    exit 1
fi

# Run the main setup
main "$@" 