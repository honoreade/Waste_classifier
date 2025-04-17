# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['multi_model_classifier.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('app_icon.ico', '.'),
        ('trained_model.h5', '.'),
        ('Garbage.h5', '.'),
        # ('final_model_weights.hdf5', '.')  # Commented out .hdf5 file
    ],
    hiddenimports=[
        'tensorflow',
        'PIL',
        'numpy',
        'customtkinter',
        'darkdetect',
        'tqdm'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MultiModelWasteClassifier1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.ico'
)