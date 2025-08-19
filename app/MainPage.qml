import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs
import QtQuick.Controls.Material 2.15



ApplicationWindow {
    id: win
    visible: true
    width: 1200
    height: 720
    title: "Zoo Classifier (Qt Quick)"

    Material.theme: Material.Dark
    Material.accent: Material.Teal

    // --------- Dosya seçici (native) ----------
    FileDialog {
        id: fileDialog
        title: "Resim Aç"
        nameFilters: ["Image files (.png *.jpg *.jpeg *.bmp)", "All files (.*)"]
        onAccepted: backend.loadImage(selectedFile)
    }

    header: ToolBar {
        RowLayout {
            anchors.fill: parent
            spacing: 10

            Button {
                text: "Resim Aç"
                onClicked: fileDialog.open()
            }

            Button {
                text: backendBusy ? "Çalışıyor..." : "Tahmin Et"
                enabled: !backendBusy
                onClicked: backend.runInference()
            }

            Button {
                text: "Klasör Tahmin"
                onClicked: folderDlg.open()
            }

            FolderDialog {
                id: folderDlg
                onAccepted: backend.runBatchOnFolder(selectedFolder)
            }

            Button {
                text: "CSV Kaydet"
                onClicked: saveDlg.open()
            }

            FileDialog {
                id: saveDlg
                title: "CSV Kaydet"
                fileMode: FileDialog.SaveFile
                nameFilters: ["CSV (*.csv)"]
                onAccepted: backend.exportCsv(selectedFile)
            }

            CheckBox {
                id: grad
                text: "Grad-CAM"
                checked: false
                onToggled: backend.setGradcam(checked)
            }

            Label { text: "Temperature" }
            Slider {
                id: temp
                from: 0.1; to: 5.0; stepSize: 0.1; value: 1.0
                Layout.preferredWidth: 200
                onValueChanged: backend.setTemperature(value)
            }
            Label { text: temp.value.toFixed(1) }

            Item { Layout.fillWidth: true }
            Label { text: statusText; elide: Text.ElideRight; Layout.preferredWidth: 400 }
        }
    }

    
    // --------- Ana içerik: sol önizleme, sağ top-5 ---------
    SplitView {
        anchors.fill: parent

        // SOL: Görsel önizleme (zoom ScrollView içinde)
        ScrollView {
            SplitView.preferredWidth: win.width * 0.6
            clip: true

            Image {
                id: preview
                anchors.fill: parent
                source: currentImage
                fillMode: Image.PreserveAspectFit   // ✅ ilk açılışta sığdır
                cache: false

                // Başlangıç scale = 1 (fit olacak şekilde)
                property real zoom: 1.0
                transform: Scale {
                    origin.x: preview.width/2
                    origin.y: preview.height/2
                    xScale: preview.zoom
                    yScale: preview.zoom
                }

                PinchHandler {
                    onScaleChanged: preview.zoom *= scale
                }
                WheelHandler {
                    onWheel: (event)=> {
                        preview.zoom *= (event.angleDelta.y > 0 ? 1.1 : 0.9)
                    }
                }
            }

        }

        // SAĞ: Sonuç listesi + görsel bilgileri
        Frame {
            SplitView.preferredWidth: win.width * 0.4
            ColumnLayout {
                anchors.fill: parent
                spacing: 8

                Label { text: "Top-5 Tahmin"; font.pixelSize: 18 }

                ListView {
                    id: resultsView
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: resultsModel
                    delegate: Column {
                        spacing: 4
                        Label { text: modelData }
                        ProgressBar {
                            from: 0; to: 100
                            value: {
                                const m = ("" + modelData).match(/([0-9]+(\.[0-9]+)?)%/);
                                return m ? parseFloat(m[1]) : 0;
                            }
                        }
                        Rectangle { height: 8; width: 1; color: "transparent" }
                    }
                }

                GroupBox {
                    title: "Görsel Bilgileri"
                    Layout.fillWidth: true
                    ColumnLayout {
                        Label { text: "Boyut: " + imageSize }
                        Label { text: "Cihaz / süre: " + deviceLatency }
                        Label { text: "Dosya: " + fileName }
                    }
                }
            }
        }
    }

    // ---------- Backend bağları ----------
    property alias backendBusy: busyFlag.checked
    property string currentImage: ""
    property var resultsModel: []
    property string statusText: ""

    property string imageSize: "-"
    property string deviceLatency: "-"
    property string fileName: "-"

    CheckBox { id: busyFlag; visible: false; checked: false }

    Connections {
        target: backend
        function onImageChanged(path) {
            currentImage = path;
        }
        function onResultsChanged(list) {
            resultsModel = list
        }
        function onBusyChanged(b) {
            busyFlag.checked = b
        }
        function onMessageChanged(msg) {
            statusText = msg
            console.log(msg)
        }
        function onMetaChanged(size, latency, file) {
            imageSize = size
            deviceLatency = latency
            fileName = file
        }
        function onInfoChanged(info) {   // ✅ yeni ekleme
        imageSize = info
        }
    }
}