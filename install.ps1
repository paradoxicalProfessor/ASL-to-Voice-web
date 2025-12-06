# Installation Guide for ASL to Voice System
# Run this script in PowerShell to set up the environment

Write-Host "=" -NoNewline
Write-Host ("=" * 69)
Write-Host "🤟 ASL to Voice - Installation Script"
Write-Host ("=" * 70)
Write-Host ""

# Check Python version
Write-Host "🔍 Checking Python installation..." -ForegroundColor Cyan
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
    
    # Check if version is 3.8+
    $version = [version]($pythonVersion -replace "Python ", "")
    if ($version.Major -lt 3 -or ($version.Major -eq 3 -and $version.Minor -lt 8)) {
        Write-Host "⚠️  Python 3.8+ required. Current version: $pythonVersion" -ForegroundColor Yellow
        Write-Host "   Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "❌ Python not found. Please install Python 3.8+" -ForegroundColor Red
    Write-Host "   Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "📋 Installation Options:" -ForegroundColor Cyan
Write-Host "1. Full installation (GPU support with CUDA)"
Write-Host "2. CPU-only installation (no CUDA required)"
Write-Host "3. Create virtual environment only"
Write-Host ""

$choice = Read-Host "Select option (1-3) [1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

# Create virtual environment
Write-Host ""
Write-Host "📦 Creating virtual environment..." -ForegroundColor Cyan

if (Test-Path "venv") {
    Write-Host "⚠️  Virtual environment already exists" -ForegroundColor Yellow
    $overwrite = Read-Host "Overwrite? (y/n) [n]"
    if ($overwrite -eq "y") {
        Remove-Item -Recurse -Force venv
        python -m venv venv
        Write-Host "✓ Virtual environment recreated" -ForegroundColor Green
    } else {
        Write-Host "Using existing virtual environment" -ForegroundColor Yellow
    }
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "   Try running: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    exit 1
}

if ($choice -eq "3") {
    Write-Host ""
    Write-Host "✅ Virtual environment created successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Activate environment: .\venv\Scripts\Activate.ps1"
    Write-Host "2. Install dependencies: pip install -r requirements.txt"
    Write-Host "3. For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    exit 0
}

# Upgrade pip
Write-Host ""
Write-Host "⬆️  Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip
Write-Host "✓ Pip upgraded" -ForegroundColor Green

# Install PyTorch
Write-Host ""
if ($choice -eq "1") {
    Write-Host "🎮 Installing PyTorch with CUDA support..." -ForegroundColor Cyan
    Write-Host "   This may take several minutes..." -ForegroundColor Yellow
    
    # Detect CUDA version (if nvidia-smi is available)
    $cudaVersion = "cu118"  # Default to CUDA 11.8
    
    try {
        $nvidiaInfo = nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ NVIDIA GPU detected" -ForegroundColor Green
            
            # Try to extract CUDA version
            if ($nvidiaInfo -match "CUDA Version: (\d+\.\d+)") {
                $detectedCuda = $matches[1]
                Write-Host "   Detected CUDA Version: $detectedCuda" -ForegroundColor Cyan
                
                # Map to PyTorch CUDA version
                if ($detectedCuda -ge "12.0") {
                    $cudaVersion = "cu121"
                } elseif ($detectedCuda -ge "11.8") {
                    $cudaVersion = "cu118"
                } elseif ($detectedCuda -ge "11.7") {
                    $cudaVersion = "cu117"
                }
            }
        }
    } catch {
        Write-Host "⚠️  Could not detect CUDA version, using default (11.8)" -ForegroundColor Yellow
    }
    
    Write-Host "   Installing PyTorch for CUDA $cudaVersion..." -ForegroundColor Cyan
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$cudaVersion"
    
} else {
    Write-Host "💻 Installing PyTorch (CPU-only)..." -ForegroundColor Cyan
    pip install torch torchvision
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ PyTorch installed" -ForegroundColor Green
} else {
    Write-Host "❌ PyTorch installation failed" -ForegroundColor Red
    exit 1
}

# Install other dependencies
Write-Host ""
Write-Host "📦 Installing other dependencies..." -ForegroundColor Cyan
Write-Host "   This may take several minutes..." -ForegroundColor Yellow

pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "❌ Some dependencies failed to install" -ForegroundColor Red
    Write-Host "   Check errors above and try installing manually" -ForegroundColor Yellow
}

# Verify installation
Write-Host ""
Write-Host "🔍 Verifying installation..." -ForegroundColor Cyan

# Test PyTorch
Write-Host ""
Write-Host "Testing PyTorch..." -ForegroundColor Cyan
$torchTest = python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host $torchTest -ForegroundColor Green
} else {
    Write-Host "⚠️  PyTorch import failed" -ForegroundColor Yellow
}

# Test other packages
$packages = @("ultralytics", "cv2", "numpy", "pyttsx3", "yaml", "matplotlib", "seaborn", "sklearn")
$allSuccess = $true

foreach ($pkg in $packages) {
    $testCmd = "import $pkg"
    $result = python -c $testCmd 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ $pkg" -ForegroundColor Green
    } else {
        Write-Host "✗ $pkg" -ForegroundColor Red
        $allSuccess = $false
    }
}

# Summary
Write-Host ""
Write-Host ("=" * 70)
if ($allSuccess) {
    Write-Host "✅ Installation completed successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Installation completed with warnings" -ForegroundColor Yellow
    Write-Host "   Some packages may need manual installation" -ForegroundColor Yellow
}
Write-Host ("=" * 70)

Write-Host ""
Write-Host "📚 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Ensure dataset is in place (train/, valid/, test/ directories)"
Write-Host "2. Run quick_start.py for interactive setup:"
Write-Host "   python quick_start.py"
Write-Host ""
Write-Host "3. Or train directly:"
Write-Host "   python train_yolov8.py"
Write-Host ""
Write-Host "4. Run live detection:"
Write-Host "   python live_inference.py"
Write-Host ""
Write-Host "📖 See README.md for detailed documentation"
Write-Host ""

# Keep terminal open
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
