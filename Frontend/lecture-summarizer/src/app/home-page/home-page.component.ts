import { Component } from '@angular/core';
import { AppService } from '../app.service';

@Component({
  selector: 'app-home-page',
  templateUrl: './home-page.component.html',
  styleUrl: './home-page.component.scss'
})
export class HomePageComponent {
  url: any;
  format: any;
  constructor(private appService: AppService) {}
  onSelectFile(event: any) {
    const file = event.target.files && event.target.files[0];
    if (file) {
      var reader = new FileReader();
      reader.readAsDataURL(file);
      if(file.type.indexOf('image')> -1){
        this.format = 'image';
      } else if(file.type.indexOf('video')> -1){
        this.format = 'video';
      }
      reader.onload = (event) => {
        this.url = (<FileReader>event.target).result;
        console.log(this.url)
      }
    }
  }

  generateText() {
    const data = {
      "video": this.url
    }
    this.appService.uploadData(data).subscribe(data => {
      console.log(data);
    })
  }
}
