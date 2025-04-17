# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['multi_model_classifier.py'],
    pathex=[],
    binaries=[],
    datas=[('trained_model.h5', '.'), ('Garbage.h5', '.'), ('final_model_weights.hdf5', '.'), ('app_icon.ico', '.')],
    hiddenimports=['tensorflow', 'PIL', 'numpy', 'customtkinter', 'darkdetect', 'tqdm'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MultiModelWasteClassifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app_icon.ico'],
)
